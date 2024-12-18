from datetime import datetime, timezone

from pydantic import BaseModel


class News(BaseModel):
    """
    This is the data model for the news.
    """

    title: str
    published_at: str  # "2024-12-18T12:29:27Z"
    source: str

    # Challenge: You can also keep the URL and scrape it to get even more context
    # about this piece of news.

    @classmethod
    def from_csv_row(
        cls,
        title: str,
        source_id: int,
        news_datetime: str,
    ) -> 'News':
        """
        This method is used to create a News object from a CSV row.

        The data we get from the CSV is in the following format:
        - title
        - sourceId
        - newsDatetime: '6/9/2022 6:57'
        """
        # parse a datetime string with this format '6/9/2022 6:57' into a datetime object
        # in UTC timezone
        news_datetime = datetime.strptime(news_datetime, '%m/%d/%Y %H:%M')
        news_datetime = news_datetime.replace(tzinfo=timezone.utc)

        # convert the datetime object into a string in the format '2024-12-18T12:29:27Z'
        published_at = news_datetime.isoformat()

        # convert the source_id into a string
        source = str(source_id)

        return cls(
            title=title,
            source=source,
            published_at=published_at,
        )

    def to_dict(self) -> dict:
        return {
            **self.model_dump(),
            'timestamp_ms': int(
                datetime.fromisoformat(
                    self.published_at.replace('Z', '+00:00')
                ).timestamp()
                * 1000
            ),
        }
